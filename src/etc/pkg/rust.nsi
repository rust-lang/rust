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

Page license
Page components
Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles


Section "Compiler"
    SetOutPath $INSTDIR
    File /oname=rustc.exe     stage3\rustc.exe
    File /oname=rustllvm.dll  stage3\rustllvm.dll
    File /oname=rustrt.dll    stage3\rustrt.dll
    File /oname=std.dll       stage3\std.dll

    SetOutPath $INSTDIR\lib
    File /oname=rustrt.dll   stage3\lib\rustrt.dll
    File /oname=std.dll      stage3\lib\std.dll
    File /oname=main.o       stage3\lib\main.o
    File /oname=glue.o       stage3\lib\glue.o
SectionEnd

Section "Documentation"
    SetOutPath $INSTDIR\doc
    File /nonfatal /oname=rust.html doc\rust.html
    File /nonfatal /oname=rust.pdf  doc\rust.pdf
SectionEnd
