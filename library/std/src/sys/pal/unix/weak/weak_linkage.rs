#[cfg(test)]
#[path = "./tests.rs"]
mod tests;

pub(crate) macro weak {
    (fn $name:ident($($param:ident : $t:ty),* $(,)?) -> $ret:ty;) => (
        let ref $name: ExternWeak<unsafe extern "C" fn($($t),*) -> $ret> = {
            unsafe extern "C" {
                #[linkage = "extern_weak"]
                static $name: Option<unsafe extern "C" fn($($t),*) -> $ret>;
            }
            #[allow(unused_unsafe)]
            ExternWeak::new(unsafe { $name })
        };
    )
}

pub(crate) struct ExternWeak<F: Copy> {
    weak_ptr: Option<F>,
}

impl<F: Copy> ExternWeak<F> {
    #[inline]
    pub fn new(weak_ptr: Option<F>) -> Self {
        ExternWeak { weak_ptr }
    }

    #[inline]
    pub fn get(&self) -> Option<F> {
        self.weak_ptr
    }
}
