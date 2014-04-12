#define CFG_VERSION_WIN GetEnv("CFG_VERSION_WIN")
#define CFG_RELEASE GetEnv("CFG_RELEASE")
#define CFG_PACKAGE_NAME GetEnv("CFG_PACKAGE_NAME")

[Setup]

SetupIconFile=rust-logo.ico
AppName=Rust
AppVersion={#CFG_RELEASE}
AppCopyright=Copyright (C) 2006-2013 Mozilla Foundation, MIT license
AppPublisher=Mozilla Foundation
AppPublisherURL=http://www.rust-lang.org
VersionInfoVersion={#CFG_VERSION_WIN}
LicenseFile=LICENSE.txt

DisableWelcomePage=true
DisableProgramGroupPage=true
DisableReadyPage=true
DisableStartupPrompt=true

OutputDir=.\dist\
SourceDir=.\
OutputBaseFilename={#CFG_PACKAGE_NAME}-install
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
Source: "tmp/dist/win/*.*" ; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

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
#include "upgrade.iss"

// Both modpath.iss and upgrade.iss want to overload CurStepChanged.
// This version does the overload then delegates to each.

procedure CurStepChanged(CurStep: TSetupStep);
begin
  UpgradeCurStepChanged(CurStep);
  ModPathCurStepChanged(CurStep);
end;
