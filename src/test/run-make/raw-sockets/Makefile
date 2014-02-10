-include ../tools.mk

ifneq ($(shell uname),Darwin)
	EXTRAFLAGS := -lm -lrt -ldl -lpthread
endif

all:
	$(RUSTC) clib.rs
	ln -s $(call STATICLIB,clib-*) $(call STATICLIB,clib)
	$(RUSTC) test.rs --test -o $(call RUN,test) -C link-args="$(call STATICLIB,clib) $(EXTRAFLAGS)"
	$(call RUN,test)
	rm $(call STATICLIB,clib*)
	rm $(call RUN,test*)
