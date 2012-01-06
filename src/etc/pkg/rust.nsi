# -*- shell-script -*-
# (not really, but syntax is similar)
#
# This is a NSIS win32 installer script the Rust toolchain.
#

Name "Rust"
ShowInstDetails "show"
ShowUninstDetails "show"
SetCompressor "lzma"
LicenseForceSelection checkbox

InstallDir $PROGRAMFILES\Rust

Page license
Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section
    WriteUninstaller $INSTDIR\uninstall.exe
SectionEnd

Section "Compiler"
    SetOutPath $INSTDIR
    File /nonfatal /r i686-pc-mingw32\stage3\*.*
SectionEnd

Section "Documentation"
    SetOutPath $INSTDIR\doc
    File /nonfatal /oname=rust.html doc\rust.html
    File /nonfatal /oname=rust.pdf  doc\rust.pdf
SectionEnd

Section "Uninstall"
    Delete $INSTDIR\uninstall.exe
    Delete $INSTDIR\bin\*.*
    Delete $INSTDIR\lib\*.*
    Delete $INSTDIR\lib\rustc\i686-pc-mingw32\bin\*.*
    Delete $INSTDIR\lib\rustc\i686-pc-mingw32\lib\*.*
    Delete $INSTDIR\doc\rust.html
    Delete $INSTDIR\doc\rust.pdf
    RMDir $INSTDIR\bin
    RMDir $INSTDIR\lib\rustc\i686-pc-mingw32\bin
    RMDir $INSTDIR\lib\rustc\i686-pc-mingw32\lib
    RMDir $INSTDIR\lib\rustc\i686-pc-mingw32
    RMDir $INSTDIR\lib\rustc
    RMDir $INSTDIR\lib
    RMDir $INSTDIR\doc
    RMDir $INSTDIR
SectionEnd
