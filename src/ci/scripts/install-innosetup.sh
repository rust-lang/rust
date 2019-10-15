#!/bin/bash
# We use InnoSetup and its `iscc` program to also create combined installers.
# Honestly at this point WIX above and `iscc` are just holdovers from
# oh-so-long-ago and are required for creating installers on Windows. I think
# one is MSI installers and one is EXE, but they're not used so frequently at
# this point anyway so perhaps it's a wash!

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    curl.exe -o is-install.exe "${MIRRORS_BASE}/2017-08-22-is.exe"
    cmd.exe //c "is-install.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP-"

    ciCommandAddPath "C:\\Program Files (x86)\\Inno Setup 5"
fi
