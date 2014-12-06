#define CFG_VERSION_WIN GetEnv("CFG_VERSION_WIN")
#define CFG_RELEASE GetEnv("CFG_RELEASE")
#define CFG_PACKAGE_NAME GetEnv("CFG_PACKAGE_NAME")
#define CFG_BUILD GetEnv("CFG_BUILD")

[Setup]

SetupIconFile=rust-logo.ico
AppName=Rust
AppVersion={#CFG_RELEASE}
AppCopyright=Copyright (C) 2006-2014 Mozilla Foundation, MIT license
AppPublisher=Mozilla Foundation
AppPublisherURL=http://www.rust-lang.org
VersionInfoVersion={#CFG_VERSION_WIN}
LicenseFile=LICENSE.txt

PrivilegesRequired=lowest
DisableWelcomePage=true
DisableProgramGroupPage=true
DisableReadyPage=true
DisableStartupPrompt=true

OutputDir=.\dist\
SourceDir=.\
OutputBaseFilename={#CFG_PACKAGE_NAME}-{#CFG_BUILD}
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

[Components]
Name: rust; Description: "Rust compiler and standard crates"; Types: full compact custom; Flags: fixed
Name: gcc; Description: "Linker and platform libraries"; Types: full

[Files]
Source: "tmp/dist/win/rust/*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs; Components: rust
Source: "tmp/dist/win/gcc/*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs; Components: gcc

[Code]
const
	ModPathName = 'modifypath';
	ModPathType = 'system';

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
