#define CFG_RELEASE_NUM GetEnv("CFG_RELEASE_NUM")
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
VersionInfoVersion={#CFG_RELEASE_NUM}
LicenseFile=LICENSE.txt

PrivilegesRequired=lowest
DisableWelcomePage=true
DisableProgramGroupPage=true
DisableReadyPage=true
DisableStartupPrompt=true

OutputDir=.\
SourceDir=.\
OutputBaseFilename={#CFG_PACKAGE_NAME}-{#CFG_BUILD}
DefaultDirName={sd}\Rust

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
#ifdef MINGW
Name: gcc; Description: "Linker and platform libraries"; Types: full
#endif
Name: docs; Description: "HTML documentation"; Types: full
Name: cargo; Description: "Cargo, the Rust package manager"; Types: full
Name: std; Description: "The Rust Standard Library"; Types: full
// tool-rls-start
Name: rls; Description: "RLS, the Rust Language Server"
// tool-rls-end

[Files]
Source: "rustc/*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs; Components: rust
#ifdef MINGW
Source: "rust-mingw/*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs; Components: gcc
#endif
Source: "rust-docs/*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs; Components: docs
Source: "cargo/*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs; Components: cargo
Source: "rust-std/*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs; Components: std
// tool-rls-start
Source: "rls/*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs; Components: rls
Source: "rust-analysis/*.*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs; Components: rls
// tool-rls-end

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
