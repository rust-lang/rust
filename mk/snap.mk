
snap-stage1: $(HSREQ1_H_$(CFG_HOST_TRIPLE))
	$(S)src/etc/make-snapshot.py $(CFG_HOST_TRIPLE)/stage1

snap-stage2: $(HSREQ2_H_$(CFG_HOST_TRIPLE)
	$(S)src/etc/make-snapshot.py $(CFG_HOST_TRIPLE)/stage2

snap-stage3: $(HSREQ3_H_$(CFG_HOST_TRIPLE)
	$(S)src/etc/make-snapshot.py $(CFG_HOST_TRIPLE)/stage3
