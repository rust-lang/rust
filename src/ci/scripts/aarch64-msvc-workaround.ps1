# Workaround for #165953.
# This should be removed once that issue is fixed.
$vcvars_ver="Preview"

# vswhere.exe is used to find Visual Studio directories.
$vswhere="${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

$setup="${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\setup.exe"
$installPath = &$vswhere -latest -property InstallationPath
Start-Process $setup -Wait -ArgumentList @(
    "modify",
    "--passive",
    "--installPath", "`"$installPath`"",
    "--add", "Microsoft.VisualStudio.Component.VC.Preview.ARM64",
    "--add", "Microsoft.VisualStudio.Component.VC.Preview.ATL.ARM64"
)

# vcvarsall.bat sets up the environment so everything should use the given versions
$vcvarsall=&$vswhere -find **\vcvarsall.bat
cd $(Split-Path -parent $vcvarsall)
# Run vcvarsall first so any errors are printed
cmd /c "vcvarsall.bat arm64 -vcvars_ver=$vcvars_ver"
# Then run it again to capture the environment and export any differences.
$old = cmd /c set
$new = cmd /c "vcvarsall.bat arm64 -vcvars_ver=$vcvars_ver > nul && set"
foreach ($entry in (Compare-Object $new $old).InputObject) {
    echo $entry >> $env:GITHUB_ENV
}
