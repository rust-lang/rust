# Free disk space on Windows GitHub action runners.

$ErrorActionPreference = 'Stop'

Get-Volume | Out-String | Write-Output

$available = $(Get-Volume C).SizeRemaining

$dirs = 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\Llvm',
'C:\rtools45', 'C:\ghcup', 'C:\Program Files (x86)\Android',
'C:\Program Files\Google\Chrome', 'C:\Program Files (x86)\Microsoft\Edge',
'C:\Program Files\Mozilla Firefox', 'C:\Program Files\MySQL', 'C:\Julia',
'C:\Program Files\MongoDB', 'C:\Program Files\Azure Cosmos DB Emulator',
'C:\Program Files\PostgreSQL', 'C:\Program Files\Unity Hub',
'C:\Strawberry', 'C:\hostedtoolcache\windows\Java_Temurin-Hotspot_jdk'

foreach ($dir in $dirs) {
    Start-ThreadJob -InputObject $dir {
        Remove-Item -Recurse -Force -LiteralPath $input
    } | Out-Null
}

foreach ($job in Get-Job) {
    Wait-Job $job  | Out-Null
    if ($job.Error) {
        Write-Output "::warning file=$PSCommandPath::$($job.Error)"
    }
    Remove-Job $job
}

Get-Volume | Out-String | Write-Output

$saved = ($(Get-Volume C).SizeRemaining - $available) / 1gb
$savedRounded = [math]::Round($saved, 3)
Write-Output "total space saved: $savedRounded GB"
