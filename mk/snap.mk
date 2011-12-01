define DEF_SNAP_FOR_STAGE_H
# $(1) stage
# $(2) triple

ifdef CFG_INSTALL_SNAP
snap-stage$(1)-H-$(2): $$(HSREQ$(1)_H_$(2))
	$(S)src/etc/make-snapshot.py stage$(1) $(2) install
else
snap-stage$(1)-H-$(2): $$(HSREQ$(1)_H_$(2))
	$(S)src/etc/make-snapshot.py stage$(1) $(2)
endif

endef

$(foreach host,$(CFG_TARGET_TRIPLES),						\
 $(eval $(foreach stage,1 2 3,								\
  $(eval $(call DEF_SNAP_FOR_STAGE_H,$(stage),$(host))))))

snap-stage1: snap-stage1-H-$(CFG_HOST_TRIPLE)

snap-stage2: snap-stage2-H-$(CFG_HOST_TRIPLE)

snap-stage3: snap-stage3-H-$(CFG_HOST_TRIPLE)