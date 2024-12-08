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
    return re.sub("\s+", " ", s)


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


def execute_command(command_interpreter, command):
    """Executes a single CLI command"""
    global new_breakpoints
    global registered_breakpoints

    res = lldb.SBCommandReturnObject()
    print(command)
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
                callback_command = (
                    "breakpoint command add -F breakpoint_callback "
                    + str(breakpoint_id)
                )
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
        print("TIMEOUT: lldb_batchmode.py has been running for too long. Aborting!")
        thread.interrupt_main()

    # Start the listener and let it run as a daemon
    watchdog_thread = threading.Thread(target=watchdog)
    watchdog_thread.daemon = True
    watchdog_thread.start()


####################################################################################################
# ~main
####################################################################################################


if len(sys.argv) != 3:
    print("usage: python lldb_batchmode.py target-path script-path")
    sys.exit(1)

target_path = sys.argv[1]
script_path = sys.argv[2]

print("LLDB batch-mode script")
print("----------------------")
print("Debugger commands script is '%s'." % script_path)
print("Target executable is '%s'." % target_path)
print("Current working directory is '%s'" % os.getcwd())

# Start the timeout watchdog
start_watchdog()

# Create a new debugger instance
debugger = lldb.SBDebugger.Create()

# When we step or continue, don't return from the function until the process
# stops. We do this by setting the async mode to false.
debugger.SetAsync(False)

# Create a target from a file and arch
print("Creating a target for '%s'" % target_path)
target_error = lldb.SBError()
target = debugger.CreateTarget(target_path, None, None, True, target_error)

if not target:
    print(
        "Could not create debugging target '"
        + target_path
        + "': "
        + str(target_error)
        + ". Aborting.",
        file=sys.stderr,
    )
    sys.exit(1)


# Register the breakpoint callback for every breakpoint
start_breakpoint_listener(target)

command_interpreter = debugger.GetCommandInterpreter()

try:
    script_file = open(script_path, "r")

    for line in script_file:
        command = line.strip()
        if (
            command == "run"
            or command == "r"
            or re.match("^process\s+launch.*", command)
        ):
            # Before starting to run the program, let the thread sleep a bit, so all
            # breakpoint added events can be processed
            time.sleep(0.5)
        if command != "":
            execute_command(command_interpreter, command)

except IOError as e:
    print("Could not read debugging script '%s'." % script_path, file=sys.stderr)
    print(e, file=sys.stderr)
    print("Aborting.", file=sys.stderr)
    sys.exit(1)
finally:
    debugger.Terminate()
    script_file.close()
