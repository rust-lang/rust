-include ../tools.mk

all:
	# Running all the shards should hit every test
	$(RUSTC) --test main.rs
	$(call RUN,main) --test-shard 1.2 | grep "test_1 ... ok"
	$(call RUN,main) --test-shard 2.2 | grep "test_2 ... ok"
