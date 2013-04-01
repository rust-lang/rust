#define CFG_VERSION GetEnv("CFG_VERSION")

[Setup]

SetupIconFile=rust-logo.ico
AppName=Rust
AppVersion={#CFG_VERSION}
AppCopyright=Copyright (C) 2006-2013 Mozilla Foundation, MIT license
AppPublisher=Mozilla Foundation
AppPublisherURL=http://www.rust-lang.org
VersionInfoVersion={#CFG_VERSION}
LicenseFile=LICENSE.txt

DisableWelcomePage=true
DisableProgramGroupPage=true
DisableReadyPage=true
DisableStartupPrompt=true

OutputDir=.\
SourceDir=.\
OutputBaseFilename=rust-{#CFG_VERSION}-install
DefaultDirName={pf32}\Rust

Compression=lzma2/ultra
InternalCompressLevel=ultra
SolidCompression=true

ChangesEnvironment=true
ChangesAssociations=no
AllowUNCPath=false
AllowNoIcons=true
Uninstallable=yes

[Tasks]
Name: modifypath; Description: &Add {app}\bin to your PATH (recommended)

[Files]
Source: "i686-pc-mingw32/stage3/*.*" ; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Code]
const
	ModPathName = 'modifypath';
	ModPathType = 'user';

function ModPathDir(): TArrayOfString;
begin
	setArrayLength(Result, 1)
	Result[0] := ExpandConstant('{app}\bin');
end;
#include "modpath.iss"