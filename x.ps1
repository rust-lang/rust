#!/usr/bin/env pwsh

# See ./x for why these scripts exist.

$ErrorActionPreference = "Stop"

# syntax check
Get-Command -syntax ${PSCommandPath} >$null

$pythonScriptArgs = @("$($PSScriptRoot)\x.py")
foreach ($arg in $args) {
  $pythonScriptArgs += """$arg"""
}

function Get-Application($app) {
  $cmd = Get-Command $app -ErrorAction SilentlyContinue -CommandType Application | Select-Object -First 1
  if ($cmd.source -match '.*AppData\\Local\\Microsoft\\WindowsApps\\.*exe') {
    # Windows for some reason puts a `python3.exe` executable in PATH that just opens the windows store.
    # Ignore it.
    return $false
  }
  return $cmd
}

function Invoke-Application($application, $arguments) {
  $process = Start-Process -NoNewWindow -PassThru $application $arguments

  # WORKAROUND: Caching the handle is necessary to make ExitCode work.
  # See https://stackoverflow.com/a/23797762
  $handle = $process.Handle
  $process.WaitForExit()

  if ($null -eq $process.ExitCode) {
    Write-Error "Unable to read the exit code"
    Exit 1
  }

  Exit $process.ExitCode
}

foreach ($python in "py", "python3", "python", "python2") {
  # NOTE: this only tests that the command exists in PATH, not that it's actually
  # executable. The latter is not possible in a portable way, see
  # https://github.com/PowerShell/PowerShell/issues/12625.
  if (Get-Application $python) {
    if ($python -eq "py") {
      # Use python3, not python2
      $pythonScriptArgs = @("-3") + $pythonScriptArgs
    }

    Invoke-Application $python $pythonScriptArgs
  }
}

$found = (Get-Application "python*" | Where-Object {$_.name -match '^python[2-3]\.[0-9]+(\.exe)?$'})
if (($null -ne $found) -and ($found.Length -ge 1)) {
  $python = $found[0]
  Invoke-Application $python $pythonScriptArgs
}

$msg = "${PSCommandPath}: error: could not find a compatible Python installation`n"
$msg += "help: consider installing it from https://www.python.org/downloads/"
Write-Error $msg -Category NotInstalled
Exit 1
