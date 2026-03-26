/* Create an object file with no `.note.GNU-stack` section.
 *
 * Assembly files do not get that section, meaning platform-default stack
 * executability is implied (usually yes on Linux).
 */

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
