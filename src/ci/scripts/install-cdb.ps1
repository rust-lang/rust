# This is a .ps1 file so we can install winget (if necessary) and easily access the registry

# Only the first three parts of the version should be specified here.
$CDB_VERSION = "10.0.22621"

# Initialize winget if it's not already initialized.
# This is needed on github's aarch64 runners.
if (!(Get-Command "winget" -ErrorAction SilentlyContinue)) {
    Add-AppxPackage -RegisterByFamilyName -MainPackage Microsoft.DesktopAppInstaller_8wekyb3d8bbwe
}

switch -Wildcard ($env:CI_JOB_NAME) {
    "*x86_64*" { $arch="x64"; break }
    "*i686*" { $arch="x86"; break }
    "*aarch64*" { $arch="arm64"; break }
    default {
        Write-Error "Unknown arch in $env:CI_JOB_NAME"
        exit 1
    }
}

$kits = Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows Kits\Installed Roots'
$kits | Get-ItemProperty | Write-Output
$debugger_path = $kits.WindowsDebuggersRoot10

$kits32 = Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\Windows Kits\Installed Roots'
Write-Output "----"
$kits32 | Get-ItemProperty | Write-Output
Write-Output "----"
if (!$debugger_path) {
    $kits = $kits32
    $debugger_path = $kits.WindowsDebuggersRoot10
}

$cdb_path = "$debugger_path\$arch\cdb.exe"
Write-Output "install path: $cdb_path"

# Try to get the existing cdb version.
# If it exists and does not match the version we want then uninstall it.
try {
    $version_str = &$cdb_path /version
    if ($version_str -match 'cdb version ([0-9]+\.[0-9]+\.[0-9]+)\..*') {
        $version = $Matches.1
        if ($version -eq $CDB_VERSION) {
            Write-Output $version_str
            exit 0
        }
    }
} catch [System.Management.Automation.CommandNotFoundException] {}

# Search the registry for the already installed debugger component
$log_file = Join-Path (Resolve-Path .) msi.log
:loop foreach ($view in [Microsoft.Win32.RegistryView]::Registry64, [Microsoft.Win32.RegistryView]::Registry32) {
    $hklm = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, $view)
    $kits = $hklm.OpenSubKey("SOFTWARE\Microsoft\Windows Kits\Installed Roots")
    foreach ($name in $kits.GetValueNames()) {
        $data = $kits.GetValue($name)
        if ($data -Like 'SDK Debuggers *') {
            $cdb_guid = $name
            # Uninstall it, if possible
            $ErrorActionPreference = 'Stop'
            if ($cdb_guid) {
                Write-Output "Uninstalling cdb ($cdb_guid)"
                Start-Process MsiExec.exe -ArgumentList "/x $cdb_guid /log $log_file /quiet IGNOREDEPENDENCIES=ALL" -Wait
            }
        }
    }
}

try {
    &$cdb_path /version
    Write-Output "uninstall failed for some reason, dumping logs"
    Get-Content $log_file | Write-Output
    #exit 1
} catch [System.Management.Automation.CommandNotFoundException] {}

# Install cdb
$ErrorActionPreference = 'Continue'
Write-Output "installing cdb..."
winget source remove msstore
winget install --id Microsoft.WindowsSDK.$CDB_VERSION `
    --source winget `
    --disable-interactivity `
    --no-upgrade `
    --accept-package-agreements `
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
$ErrorActionPreference = 'Stop'
#&$cdb_path /version

$version_str = &$cdb_path /version
if ($version_str -match 'cdb version ([0-9]+\.[0-9]+\.[0-9]+)\..*') {
    $version = $Matches.1
    if ($version -eq $CDB_VERSION) {
        exit 0
    } else {
        Write-Output "wrong cdb version"
        Write-Output "expected: $CDB_VERSION"
        Write-Output "found: $version"
    }
} else {
    Write-Error "failed to read cdb version string: $version_str"
}

exit 1
