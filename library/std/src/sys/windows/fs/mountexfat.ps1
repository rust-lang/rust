param(
	[Parameter(Mandatory=$true)]
	[String]$vhdPath
)

$ErrorActionPreference = "Stop"

if (Test-Path -LiteralPath $vhdPath) {
	# Get the already created volume.
	$vhd = Get-VHD -Path $vhdPath
	# If it's not already mounted, mount the vhd.
	if ($vhd.Attached -eq $false) {
		$disk = Mount-VHD -Path $vhd.Path -PassThru | Get-Disk
	}
	$volume = $disk | Get-Partition | Get-Volume
} else {
	# Create and setup the disk
	$vhd = New-VHD -Path $vhdPath -SizeBytes 3MB
	$disk = Mount-VHD -Path $vhd.Path -PassThru
	Initialize-Disk $disk.DiskNumber
	$partition = New-Partition -AssignDriveLetter -UseMaximumSize -DiskNumber $disk.DiskNumber
	$volume = Format-Volume -FileSystem exFAT -Partition $partition -Confirm:$false -Force
}

# Return a single letter (A-Z) where the drive is mounted.
return $volume.DriveLetter
