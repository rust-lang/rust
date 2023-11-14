#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"

# Syntax check
Get-Command -Syntax ${PSCommandPath} >$null

# Path to the x.py script
$xpy = Join-Path $PSScriptRoot x.py

# Define the arguments for x.py
$xpy_args = @($xpy)

# Quote arguments to ensure they are passed correctly
foreach ($arg in $args) {
    $xpy_args += '"' + $arg + '"'
}

# Function to get an application's command
function Get-Application($app) {
    Get-Command $app -ErrorAction SilentlyContinue -CommandType Application | Select-Object -First 1
}

# Function to invoke an application
function Invoke-Application($application, $arguments) {
    $process = Start-Process -NoNewWindow -PassThru $application $arguments
    $process.WaitForExit()
    if ($null -eq $process.ExitCode) {
        Write-Error "Unable to read the exit code"
        Exit 1
    }
    Exit $process.ExitCode
}

# Try various Python interpreters
foreach ($python in "py", "python3", "python", "python2") {
    if ($app = Get-Application $python) {
        if ($python -eq "py") {
            # Use python3 instead of python2
            $xpy_args = @("-3") + $xpy_args
        }
        Invoke-Application $app $xpy_args
    }
}

# Try to find a Python interpreter (version 2 or 3) in PATH
$found = Get-Application "python*" | Where-Object {$_.Name -match '^python[2-3]\.[0-9]+(\.exe)?$'}
if ($found -ne $null -and $found.Length -ge 1) {
    $python = $found[0]
    Invoke-Application $python $xpy_args
}

# Display an error message if Python is not found
$msg = "${PSCommandPath}: error: Python is not installed`n"
$msg += "help: Consider installing Python from https://www.python.org/downloads/"
Write-Error $msg -Category NotInstalled
Exit 1
