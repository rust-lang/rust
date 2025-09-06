/* Device specific memory layout */

/* This file is used to build the cortex-m-rt examples,
   but not other applications using cortex-m-rt. */

MEMORY
{
  /* FLASH and RAM are mandatory memory regions */
  /* Update examples/data_overflow.rs if you change these sizes. */
  FLASH : ORIGIN = 0x00000000, LENGTH = 256K
  RAM : ORIGIN = 0x20000000, LENGTH = 64K

  /* More memory regions can declared: for example this is a second RAM region */
  /* CCRAM : ORIGIN = 0x10000000, LENGTH = 8K */
}

/* The location of the stack can be overridden using the `_stack_start` symbol.
   By default it will be placed at the end of the RAM region */
/* _stack_start = ORIGIN(CCRAM) + LENGTH(CCRAM); */

/* The location of the .text section can be overridden using the `_stext` symbol.
   By default it will place after .vector_table */
/* _stext = ORIGIN(FLASH) + 0x40c; */