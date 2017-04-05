# Copyright 2015-2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

from __future__ import print_function
import datetime
import sys

def format_build_time(duration):
    return str(datetime.timedelta(seconds=int(duration)))

def notify_linux(title, text):
    try:
        import dbus
        bus = dbus.SessionBus()
        notify_obj = bus.get_object("org.freedesktop.Notifications",
                                    "/org/freedesktop/Notifications")
        method = notify_obj.get_dbus_method("Notify", "org.freedesktop.Notifications")
        method(title, 0, "", text, "", [], {"transient": True}, -1)
    except:
        raise Exception("Optional Python module 'dbus' is not installed.")


def notify_win(title, text):
    try:
        from servo.win32_toast import WindowsToast
        w = WindowsToast()
        w.balloon_tip(title, text)
    except:
        from ctypes import Structure, windll, POINTER, sizeof
        from ctypes.wintypes import DWORD, HANDLE, WINFUNCTYPE, BOOL, UINT

        class FLASHWINDOW(Structure):
            _fields_ = [("cbSize", UINT),
                        ("hwnd", HANDLE),
                        ("dwFlags", DWORD),
                        ("uCount", UINT),
                        ("dwTimeout", DWORD)]

        FlashWindowExProto = WINFUNCTYPE(BOOL, POINTER(FLASHWINDOW))
        FlashWindowEx = FlashWindowExProto(("FlashWindowEx", windll.user32))
        FLASHW_CAPTION = 0x01
        FLASHW_TRAY = 0x02
        FLASHW_TIMERNOFG = 0x0C

        params = FLASHWINDOW(sizeof(FLASHWINDOW),
                             windll.kernel32.GetConsoleWindow(),
                             FLASHW_CAPTION | FLASHW_TRAY | FLASHW_TIMERNOFG, 3, 0)
        FlashWindowEx(params)


def notify_darwin(title, text):
    try:
        import Foundation

        bundleDict = Foundation.NSBundle.mainBundle().infoDictionary()
        bundleIdentifier = 'CFBundleIdentifier'
        if bundleIdentifier not in bundleDict:
            bundleDict[bundleIdentifier] = 'mach'

        note = Foundation.NSUserNotification.alloc().init()
        note.setTitle_(title)
        note.setInformativeText_(text)

        now = Foundation.NSDate.dateWithTimeInterval_sinceDate_(0, Foundation.NSDate.date())
        note.setDeliveryDate_(now)

        centre = Foundation.NSUserNotificationCenter.defaultUserNotificationCenter()
        centre.scheduleNotification_(note)
    except ImportError as e:
        if str(e) == 'No module named Foundation':
            print('Try running: `python2.7 -m pip install pyobjc-framework-AVFoundation`')
        raise Exception("Optional Python module 'pyobjc' is not installed.")


def notify_with_command(command):
    def notify(title, text):
        if call([command, title, text]) != 0:
            raise Exception("Could not run '%s'." % command)
    return notify


def notify(title, text):
    """Generate a desktop notification using appropriate means on
    supported platforms Linux, Windows, and Mac OS.  On unsupported
    platforms, this function acts as a no-op.
    """
    platforms = {
        "linux": notify_linux,
        "linux2": notify_linux,
        "win32": notify_win,
        "darwin": notify_darwin
    }
    func = platforms.get(sys.platform)

    if func is not None:
        try:
            func(title, text)
        except Exception as e:
            extra = getattr(e, "message", "")
            print("[Warning] Could not generate notification! %s" % extra, file=sys.stderr)


def notify_build_done(elapsed, success=True):
    notify("Rust build",
           "%s in %s" % ("Completed" if success else "FAILED", format_build_time(elapsed)))
