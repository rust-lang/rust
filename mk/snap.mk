
snap-stage1: $(HSREQ1_H_$(CFG_HOST_TRIPLE))
	$(S)src/etc/make-snapshot.py stage1 $(CFG_HOST_TRIPLE)

snap-stage2: $(HSREQ2_H_$(CFG_HOST_TRIPLE)
	$(S)src/etc/make-snapshot.py stage2 $(CFG_HOST_TRIPLE)

snap-stage3: $(HSREQ3_H_$(CFG_HOST_TRIPLE)
	$(S)src/etc/make-snapshot.py stage3 $(CFG_HOST_TRIPLE)
