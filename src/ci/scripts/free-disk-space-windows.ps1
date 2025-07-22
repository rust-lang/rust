# Free disk space on Windows GitHub action runners.

$ErrorActionPreference = 'Stop'

Get-Volume | Out-String | Write-Output

$available = $(Get-Volume C).SizeRemaining

$dirs = 'Does not exit',
'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\Llvm',
'C:\rtools45', 'C:\ghcup', 'C:\Program Files (x86)\Android',
'C:\Program Files\Google\Chrome', 'C:\Program Files (x86)\Microsoft\Edge',
'C:\Program Files\Mozilla Firefox', 'C:\Program Files\MySQL', 'C:\Julia',
'C:\Program Files\MongoDB', 'C:\Program Files\Azure Cosmos DB Emulator',
'C:\Program Files\PostgreSQL', 'C:\Program Files\Unity Hub',
'C:\Strawberry', 'C:\hostedtoolcache\windows\Java_Temurin-Hotspot_jdk',
'C:\does not exist', 'oh no'

foreach ($dir in $dirs) {
    Start-ThreadJob { Remove-Item -Recurse -Force $dir }
}

foreach ($job in Get-Job) {
    Wait-Job $job
    # Print if any, any.
    Write-Output "::warning file=$PSCommandPath::$($job.Error)"
    Remove-Job $job
}

Get-Volume | Out-String | Write-Output

$saved = ($(Get-Volume C).SizeRemaining - $available) / 1gb
Write-Output "total space saved $saved GB"
