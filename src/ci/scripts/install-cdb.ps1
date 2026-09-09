# Change the version below to pin a different version of cdb.exe
# Note: only the first three parts of the version should be specified here.
$CDB_VERSION = "10.0.22621"

# This script installs the given version of cdb.
# Unfortunately this is more complext than it could be due to limitations of using winsdksetup.exe via the CLI.
# We need to uninstall the old version (using the MSI installer instead of winsdksetup.exe)
# then install the version we want.
# Further complicating this are problem with the way Github's aarch64 runners are (or aren't) setup.
#
# If this script is breaking CI, feel free to disable it and skip any failing debug tests until
# someone comes a long to fix.

switch -Wildcard ($env:CI_JOB_NAME) {
    "*x86_64*" { $arch="x64"; break }
    "*i686*" { $arch="x86"; break }
    "*aarch64*" { $arch="arm64"; break }
    default {
        Write-Error "Unknown arch in $env:CI_JOB_NAME"
        exit 1
    }
}

# Presumably for legacy reasons, the registry values we need are split between the 64-bit and 32-bit registries.
# In order to guard against future changes, we always check both 64-bit and 32-bit registries even
# if we know a particular value is currently in the 32-bit registry.
$kits_path = "SOFTWARE\Microsoft\Windows Kits\Installed Roots"
$hklm64 = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry64)
$hklm32 = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry32)
$kits64 = $hklm64.OpenSubKey($kits_path)
$kits32 = $hklm32.OpenSubKey($kits_path)

$debugger_path = $kits64.GetValue("WindowsDebuggersRoot10")
if (!$debugger_path) {
    $debugger_path = $kits32.GetValue("WindowsDebuggersRoot10")
}

$cdb_path = "$debugger_path\$arch\cdb.exe"
Write-Output "install path: $cdb_path"

# Get the existing cdb version.
# If it matches the version we want then we can exit this script early.
$version_str = &$cdb_path /version
if ($version_str -match 'cdb version ([0-9]+\.[0-9]+\.[0-9]+)\..*') {
    $version = $Matches.1
    if ($version -eq $CDB_VERSION) {
        Write-Output $version_str
        exit 0
    }
}

# Search the registry for the already installed debugger component and uninstall it.
:loop foreach ($kits in $kits64, $kits32) {
    foreach ($name in $kits.GetValueNames()) {
        $data = $kits.GetValue($name)
        if ($data -Like 'SDK Debuggers *') {
            $cdb_guid = $name
            # Uninstall it, if possible
            if ($cdb_guid) {
                Write-Output "Uninstalling cdb ($cdb_guid)"
                Start-Process MsiExec.exe -ArgumentList "/x $cdb_guid /quiet IGNOREDEPENDENCIES=ALL" -Wait
                break loop
            }
        }
    }
}

try {
    &$cdb_path /version
    # If the uninstall worked properly then we won't reach this line
    # However, on GitHub's aarch64 runners the uninstaller is borked.
    # It reports success and removes the registry entries but leaves the files.
    Remove-Item -recurse -force $debugger_path
} catch [System.Management.Automation.CommandNotFoundException] {}

# Initialize winget if it's not already initialized.
# This is again needed for GitHub's aarch64 runners.
if (!(Get-Command "winget" -ErrorAction SilentlyContinue)) {
    Add-AppxPackage -RegisterByFamilyName -MainPackage Microsoft.DesktopAppInstaller_8wekyb3d8bbwe
}
$ErrorActionPreference = 'Continue'
# msstore won't work in CI but will make winget flaky on aarch64 even if explicitly using a different source
winget source remove msstore

Write-Output "installing cdb..."
# We use `--force` to make the installer run even if this version is already installed
# so that we can add the debugger feature.
winget install --id Microsoft.WindowsSDK.$CDB_VERSION `
    --source winget `
    --disable-interactivity `
    --no-upgrade `
    --force `
    --override "/q /features OptionId.WindowsDesktopDebuggers"
$result = $LastExitCode

# If already installed then don't error, otherwise exit with the error if any
# For error codes see:
# https://github.com/microsoft/winget-cli/blob/master/doc/windows/package-manager/winget/returnCodes.md
$PACKAGE_ALREADY_INSTALLED = 0x8A150061
if ( $result -notin 0,$PACKAGE_ALREADY_INSTALLED )
{
    Write-Output "winget failed with exit code: $result"
    exit $result
}

# Print the installed cdb version
$version_str = &$cdb_path /version
Write-Output $version_str
if ($version_str -match 'cdb version ([0-9]+\.[0-9]+\.[0-9]+)\..*') {
    $version = $Matches.1
    if ($version -eq $CDB_VERSION) {
        # Success!
        exit 0
    } else {
        Write-Output "wrong cdb version"
        Write-Output "expected $CDB_VERSION"
        Write-Output "found $version"
    }
} else {
    Write-Error "failed to read cdb version string: $version_str"
}

exit 1
