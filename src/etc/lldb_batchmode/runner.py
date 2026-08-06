# This script allows to use LLDB in a way similar to GDB's batch mode. That is, given a text file
# containing LLDB commands (one command per line), this script will execute the commands one after
# the other.
# LLDB also has the -s and -S commandline options which also execute a list of commands from a text
# file. However, this command are execute `immediately`: the command of a `run` or `continue`
# command will be executed immediately after the `run` or `continue`, without waiting for the next
# breakpoint to be hit. This a command sequence like the following will not yield reliable results:
#
#   break 11
#   run
#   print x
#
# Most of the time the `print` command will be executed while the program is still running will thus
# fail. Using this Python script, the above will work as expected.

from __future__ import print_function
import lldb
import os
import sys
import threading
import re
import time
import traceback


try:
    import thread
except ModuleNotFoundError:
    # The `thread` module was renamed to `_thread` in Python 3.
    import _thread as thread

# Set this to True for additional output
DEBUG_OUTPUT = True


def print_debug(s):
    """Print something if DEBUG_OUTPUT is True"""
    global DEBUG_OUTPUT
    if DEBUG_OUTPUT:
        print("DEBUG: " + str(s))


def normalize_whitespace(s):
    """Replace newlines, tabs, multiple spaces, etc with exactly one space"""
    return re.sub(r"\s+", " ", s)


def breakpoint_callback(frame, bp_loc, dict):
    """This callback is registered with every breakpoint and makes sure that the
    frame containing the breakpoint location is selected"""

    # HACK(eddyb) print a newline to avoid continuing an unfinished line.
    print("")
    print("Hit breakpoint " + str(bp_loc))

    # Select the frame and the thread containing it
    frame.thread.process.SetSelectedThread(frame.thread)
    frame.thread.SetSelectedFrame(frame.idx)

    # Returning True means that we actually want to stop at this breakpoint
    return True


# This is a list of breakpoints that are not registered with the breakpoint callback. The list is
# populated by the breakpoint listener and checked/emptied whenever a command has been executed
new_breakpoints = []

# This set contains all breakpoint ids that have already been registered with a callback, and is
# used to avoid hooking callbacks into breakpoints more than once
registered_breakpoints = set()


def execute_command(command_interpreter: lldb.SBCommandInterpreter, command: str):
    """Executes a single CLI command"""
    global new_breakpoints
    global registered_breakpoints

    res = lldb.SBCommandReturnObject()
    print(f"(lldb) {command}")
    command_interpreter.HandleCommand(command, res)

    if res.Succeeded():
        if res.HasResult():
            print(normalize_whitespace(res.GetOutput() or ""), end="\n")

        # If the command introduced any breakpoints, make sure to register
        # them with the breakpoint
        # callback
        while len(new_breakpoints) > 0:
            res.Clear()
            breakpoint_id = new_breakpoints.pop()

            if breakpoint_id in registered_breakpoints:
                print_debug(
                    "breakpoint with id %s is already registered. Ignoring."
                    % str(breakpoint_id)
                )
            else:
                print_debug(
                    "registering breakpoint callback, id = " + str(breakpoint_id)
                )
                callback_command = f"breakpoint command add -s python {str(breakpoint_id)} -o \
'import lldb_batchmode; lldb_batchmode.runner.breakpoint_callback'"

                command_interpreter.HandleCommand(callback_command, res)
                if res.Succeeded():
                    print_debug(
                        "successfully registered breakpoint callback, id = "
                        + str(breakpoint_id)
                    )
                    registered_breakpoints.add(breakpoint_id)
                else:
                    print(
                        "Error while trying to register breakpoint callback, id = "
                        + str(breakpoint_id)
                        + ", message = "
                        + str(res.GetError())
                    )
    else:
        print(res.GetError())


def start_breakpoint_listener(target):
    """Listens for breakpoints being added and adds new ones to the callback
    registration list"""
    listener = lldb.SBListener("breakpoint listener")

    def listen():
        event = lldb.SBEvent()
        try:
            while True:
                if listener.WaitForEvent(120, event):
                    if (
                        lldb.SBBreakpoint.EventIsBreakpointEvent(event)
                        and lldb.SBBreakpoint.GetBreakpointEventTypeFromEvent(event)
                        == lldb.eBreakpointEventTypeAdded
                    ):
                        global new_breakpoints
                        breakpoint = lldb.SBBreakpoint.GetBreakpointFromEvent(event)
                        print_debug("breakpoint added, id = " + str(breakpoint.id))
                        new_breakpoints.append(breakpoint.id)
        except BaseException:  # explicitly catch ctrl+c/sysexit
            print_debug("breakpoint listener shutting down")

    # Start the listener and let it run as a daemon
    listener_thread = threading.Thread(target=listen)
    listener_thread.daemon = True
    listener_thread.start()

    # Register the listener with the target
    target.GetBroadcaster().AddListener(
        listener, lldb.SBTarget.eBroadcastBitBreakpointChanged
    )


def start_watchdog():
    """Starts a watchdog thread that will terminate the process after a certain
    period of time"""

    try:
        from time import clock
    except ImportError:
        from time import perf_counter as clock

    watchdog_start_time = clock()
    watchdog_max_time = watchdog_start_time + 30

    def watchdog():
        while clock() < watchdog_max_time:
            time.sleep(1)
        print("TIMEOUT: lldb_batchmode has been running for too long. Aborting!")
        thread.interrupt_main()

    # Start the listener and let it run as a daemon
    watchdog_thread = threading.Thread(target=watchdog)
    watchdog_thread.daemon = True
    watchdog_thread.start()


def get_env_arg(name):
    value = os.environ.get(name)
    if value is None:
        print("must set %s" % name)
        sys.exit(1)
    return value


def dispatch_repr(var_name: str, breakpoint_index: int, frame: lldb.SBFrame) -> bool:
    # We save importing the check until we actually see a repr command. This prevents us from trying
    # to load input data from tests that don't use `repr` commands.
    from .check_lldb import check
    from .common import Result

    return check(var_name, breakpoint_index, frame) == Result.Ok


####################################################################################################
# ~main
####################################################################################################


def main():
    target_path = get_env_arg("LLDB_BATCHMODE_TARGET_PATH")
    script_path = get_env_arg("LLDB_BATCHMODE_SCRIPT_PATH")

    print("LLDB batch-mode script")
    print("----------------------")
    print(f"Python version: {sys.version}")
    print("Debugger commands script is '%s'." % script_path)
    print("Target executable is '%s'." % target_path)
    print("Current working directory is '%s'" % os.getcwd())

    # Start the timeout watchdog
    start_watchdog()

    # This is the debugger instance of the lldb executable that imported and ran this python script.
    # There is some weird behavior around LLDB reassigning, clearing, or not updating their own
    # references (like `lldb.debugger`) while a python function is actively running (i.e. if control
    # is not given back to the REPL). To prevent LLDB from changing things out from under us, we
    # store this reference locally.
    debugger = lldb.debugger

    # When we step or continue, don't return from the function until the process
    # stops. We do this by setting the async mode to false.
    debugger.SetAsync(False)

    # Create a target from a file and arch
    print("Creating a target for '%s'" % target_path)

    target: lldb.SBTarget = debugger.CreateTargetWithFileAndTargetTriple(
        target_path, lldb.SBPlatform.GetHostPlatform().GetTriple()
    )

    if not target or not target.IsValid():
        print(
            "Could not create debugging target '" + target_path + ". Aborting.",
        )
        sys.exit(1)

    # Register the breakpoint callback for every breakpoint
    start_breakpoint_listener(target)

    command_interpreter = debugger.GetCommandInterpreter()

    repr_cmd_run = False
    breakpoint_index = 0
    all_ok = True

    try:
        script_file = open(script_path, "r")

        for line in script_file:
            command = line.strip()
            if (
                command == "run"
                or command == "r"
                or re.match(r"^process\s+launch.*", command)
            ):
                print(f"(lldb) {command}")
                process: lldb.SBProcess = target.LaunchSimple(None, None, None)
                if (
                    process.GetSelectedThread().GetStopReason()
                    == lldb.eStopReasonBreakpoint
                    and breakpoint_index is None
                ):
                    breakpoint_index = 0
                continue
            if command == "continue" or command == "c":
                print(f"(lldb) {command}")
                process.Continue()
                if (
                    process.GetSelectedThread().GetStopReason()
                    == lldb.eStopReasonBreakpoint
                ):
                    breakpoint_index += 1
                continue
            if command == "quit" or command == "exit":
                print(f"(lldb) {command}")
                break
            if command.startswith("repr "):
                repr_cmd_run = True
                var_name = command.split(" ", 1)[1]

                p = target.GetProcess()
                frame = p.GetSelectedThread().GetSelectedFrame()

                print(command)
                all_ok &= dispatch_repr(var_name, breakpoint_index, frame)
            elif command != "":
                execute_command(command_interpreter, command)

    except IOError as e:
        print("Could not read debugging script '%s'." % script_path)
        traceback.print_exception(type(e), e, e.__traceback__, file=sys.stdout)
        print("Aborting.")
        # Returning status codes using `sys.exit` doesn't work since we're in an LLDB managed python
        # instance. This command sets the exit code but *does not* kill LLDB, the debugee process,
        # or the SBDebugger object.
        debugger.HandleCommand("quit 1")
    except Exception as e:
        traceback.print_exception(e, file=sys.stdout)
        debugger.HandleCommand("quit 1")
    else:  # Executes if the `try` block throws no exceptions.
        if repr_cmd_run:
            # We save importing these until we actually see a repr command. This prevents us
            # from trying to load input data from tests that don't use `repr` commands.
            from .check_lldb import tested_all_types, tested_all_variables
            from .common import BLESS, BlessMetadata, INPUT_DATA

            # `bless` should resolve any errors from mismatched test data, so any errors that reach
            # this point are either from the `bless` not working properly, or some other issue with
            # the test itself. In either case, we probably don't want to update the test data until
            # those are resolved.
            # Only runs if the test contains a repr command, as we don't want to create an input
            # file for a test that won't ever use it.

            if not tested_all_types() or not tested_all_variables():
                debugger.HandleCommand("quit 1")
            elif BLESS:
                from lldb_providers import FEATURE_FLAGS

                INPUT_DATA.save_blessing(
                    BlessMetadata(
                        sys.version, debugger.GetVersionString(), str(FEATURE_FLAGS)
                    )
                )
    finally:
        script_file.close()
