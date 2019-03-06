Write-Host "Executing:" $args[0] "..."
sh $args[0]
Write-Host "Command:" $args[0] "exited with error code" $LASTEXITCODE
Write-Host "Done."
