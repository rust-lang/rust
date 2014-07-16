# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# This script allows to use LLDB in a way similar to GDB's batch mode. That is, given a text file
# containing LLDB commands (one command per line), this script will execute the commands one after
# the other.
# LLDB also has the -s and -S commandline options which also execute a list of commands from a text
# file. However, this command are execute `immediately`: a the command of a `run` or `continue`
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
import atexit

# Set this to True for additional output
DEBUG_OUTPUT = False

def print_debug(s):
  "Print something if DEBUG_OUTPUT is True"
  global DEBUG_OUTPUT
  if DEBUG_OUTPUT:
    print("DEBUG: " + str(s))


def normalize_whitespace(s):
  "Replace newlines, tabs, multiple spaces, etc with exactly one space"
  return re.sub("\s+", " ", s)


# This callback is registered with every breakpoint and makes sure that the frame containing the
# breakpoint location is selected
def breakpoint_callback(frame, bp_loc, dict):
  "Called whenever a breakpoint is hit"
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
  "Executes a single CLI command"
  global new_breakpoints
  global registered_breakpoints

  res = lldb.SBCommandReturnObject()
  print(command)
  command_interpreter.HandleCommand(command, res)

  if res.Succeeded():
      if res.HasResult():
          print(normalize_whitespace(res.GetOutput()), end = '\n')

      # If the command introduced any breakpoints, make sure to register them with the breakpoint
      # callback
      while len(new_breakpoints) > 0:
        res.Clear()
        breakpoint_id = new_breakpoints.pop()

        if breakpoint_id in registered_breakpoints:
          print_debug("breakpoint with id %s is already registered. Ignoring." % str(breakpoint_id))
        else:
          print_debug("registering breakpoint callback, id = " + str(breakpoint_id))
          callback_command = "breakpoint command add -F breakpoint_callback " + str(breakpoint_id)
          command_interpreter.HandleCommand(callback_command, res)
          if res.Succeeded():
            print_debug("successfully registered breakpoint callback, id = " + str(breakpoint_id))
            registered_breakpoints.add(breakpoint_id)
          else:
            print("Error while trying to register breakpoint callback, id = " + str(breakpoint_id))
  else:
      print(res.GetError())


def start_breakpoint_listener(target):
  "Listens for breakpoints being added and adds new ones to the callback registration list"
  listener = lldb.SBListener("breakpoint listener")

  def listen():
    event = lldb.SBEvent()
    try:
      while True:
        if listener.WaitForEvent(120, event):
          if lldb.SBBreakpoint.EventIsBreakpointEvent(event) and \
             lldb.SBBreakpoint.GetBreakpointEventTypeFromEvent(event) == \
             lldb.eBreakpointEventTypeAdded:
            global new_breakpoints
            breakpoint = lldb.SBBreakpoint.GetBreakpointFromEvent(event)
            print_debug("breakpoint added, id = " + str(breakpoint.id))
            new_breakpoints.append(breakpoint.id)
    except:
      print_debug("breakpoint listener shutting down")

  # Start the listener and let it run as a daemon
  listener_thread = threading.Thread(target = listen)
  listener_thread.daemon = True
  listener_thread.start()

  # Register the listener with the target
  target.GetBroadcaster().AddListener(listener, lldb.SBTarget.eBroadcastBitBreakpointChanged)


####################################################################################################
# ~main
####################################################################################################

if len(sys.argv) != 3:
  print("usage: python lldb_batchmode.py target-path script-path")
  sys.exit(1)

target_path = sys.argv[1]
script_path = sys.argv[2]


# Create a new debugger instance
debugger = lldb.SBDebugger.Create()

# When we step or continue, don't return from the function until the process
# stops. We do this by setting the async mode to false.
debugger.SetAsync(False)

# Create a target from a file and arch
print("Creating a target for '%s'" % target_path)
target = debugger.CreateTargetWithFileAndArch(target_path, lldb.LLDB_ARCH_DEFAULT)

if not target:
  print("Could not create debugging target '" + target_path + "'. Aborting.", file=sys.stderr)
  sys.exit(1)


# Register the breakpoint callback for every breakpoint
start_breakpoint_listener(target)

command_interpreter = debugger.GetCommandInterpreter()

try:
  script_file = open(script_path, 'r')

  for line in script_file:
    command = line.strip()
    if command != '':
      execute_command(command_interpreter, command)

except IOError as e:
  print("Could not read debugging script '%s'." % script_path, file = sys.stderr)
  print(e, file = sys.stderr)
  print("Aborting.", file = sys.stderr)
  sys.exit(1)
finally:
  script_file.close()

