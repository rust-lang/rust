include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) --print uwu 2>&1 | diff - valid-print-requests.stderr
