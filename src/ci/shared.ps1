function retry
{
    param (
    [Parameter(Mandatory=$true)][string]$Command    
    )

    $retrycount = 1
    $max = 5
    $completed = $false

    while ((-not $completed) -and ($retrycount -le 5)) {
        
            & { $command }

            if ($LASTEXITCODE -ne 0) {
                $retrycount += 1
                start-sleep -Seconds 2
                Write-host "Command failed.  Attempt $retrycount/$max"
            }
            else {
                $completed = $true
            }
    }
    if (-not $completed) {
        Write-Host "Failed after $max attempts."
        throw
    }
}

