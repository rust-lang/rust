
snap-stage1: $(HOST_SREQ1)
	$(S)src/etc/make-snapshot.py stage1

snap-stage2: $(HOST_SREQ2)
	$(S)src/etc/make-snapshot.py stage2

snap-stage3: $(HOST_SREQ3)
	$(S)src/etc/make-snapshot.py stage3
