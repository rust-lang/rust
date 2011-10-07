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
    SetOutPath $INSTDIR\bin
    File /oname=rustc.exe      stage3\bin\rustc.exe

    SetOutPath $INSTDIR\lib
    File /oname=rustllvm.dll   stage3\lib\rustllvm.dll
    File /oname=rustrt.dll     stage3\lib\rustrt.dll
    File /oname=std.dll        stage3\lib\std.dll

    SetOutPath $INSTDIR\lib\rustc\i686-pc-mingw32\lib
    File /oname=rustrt.dll    stage3\lib\rustc\i686-pc-mingw32\lib\rustrt.dll
    File /oname=std.dll       stage3\lib\rustc\i686-pc-mingw32\lib\std.dll
    File /oname=main.o        stage3\lib\rustc\i686-pc-mingw32\lib\main.o
    File /oname=intrinsics.bc stage3\lib\rustc\i686-pc-mingw32\lib\intrinsics.bc
SectionEnd

Section "Documentation"
    SetOutPath $INSTDIR\doc
    File /nonfatal /oname=rust.html doc\rust.html
    File /nonfatal /oname=rust.pdf  doc\rust.pdf
SectionEnd

Section "Uninstall"
    Delete $INSTDIR\uninstall.exe
    Delete $INSTDIR\bin\rustc.exe
    Delete $INSTDIR\lib\rustllvm.dll
    Delete $INSTDIR\lib\rustrt.dll
    Delete $INSTDIR\lib\std.dll
    Delete $INSTDIR\lib\rustc\i686-pc-mingw32\lib\rustrt.dll
    Delete $INSTDIR\lib\rustc\i686-pc-mingw32\lib\std.dll
    Delete $INSTDIR\lib\rustc\i686-pc-mingw32\lib\main.o
    Delete $INSTDIR\lib\rustc\i686-pc-mingw32\lib\intrinsics.bc
    Delete $INSTDIR\doc\rust.html
    Delete $INSTDIR\doc\rust.pdf
    RMDir $INSTDIR\bin
    RMDir $INSTDIR\lib\rustc\i686-pc-mingw32\lib
    RMDir $INSTDIR\lib\rustc\i686-pc-mingw32
    RMDir $INSTDIR\lib\rustc
    RMDir $INSTDIR\lib
    RMDir $INSTDIR\doc
    RMDir $INSTDIR
SectionEnd