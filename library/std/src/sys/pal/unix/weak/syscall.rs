use super::weak;

pub(crate) macro syscall {
    (
        fn $name:ident($($param:ident : $t:ty),* $(,)?) -> $ret:ty;
    ) => (
        unsafe fn $name($($param: $t),*) -> $ret {
            weak!(fn $name($($param: $t),*) -> $ret;);

            // Use a weak symbol from libc when possible, allowing `LD_PRELOAD`
            // interposition, but if it's not found just use a raw syscall.
            if let Some(fun) = $name.get() {
                unsafe { fun($($param),*) }
            } else {
                unsafe { libc::syscall(libc::${concat(SYS_, $name)}, $($param),*) as $ret }
            }
        }
    )
}
