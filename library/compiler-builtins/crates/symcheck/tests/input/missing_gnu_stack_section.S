/* Create an object file with no `.note.GNU-stack` section.
 *
 * Assembly files do not get that section, meaning platform-default stack
 * executability is implied (usually yes on Linux).
 */

/* Unless we set the ABI version, we have no way to differentiate between
   big-endian ppc64 ELFv1 or ELFv2 since e_flags is otherwise unspecified */
#if defined(__powerpc64__) && _CALL_ELF == 2
.abiversion 2
#endif

.global func

#ifdef __wasm__
.functype func () -> ()
.type     func, @function
#endif

func:
    nop

#ifdef __wasm__
    end_function
#endif
