#!/bin/sh

prefix=@prefix@
exec_prefix=@exec_prefix@
libdir=@libdir@

@LD_PRELOAD_VAR@=${libdir}/libjemalloc.@SOREV@
export @LD_PRELOAD_VAR@
exec "$@"
