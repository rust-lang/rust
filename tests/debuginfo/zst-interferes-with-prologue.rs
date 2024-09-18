//@ min-lldb-version: 310

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:break zst_interferes_with_prologue::Foo::var_return_opt_try
// gdb-command:run

// gdb-command:print self
// gdb-command:next
// gdb-command:print self
// gdb-command:print $1 == $2
// gdb-check:true

// === LLDB TESTS ==================================================================================

// lldb-command:b "zst_interferes_with_prologue::Foo::var_return_opt_try"
// lldb-command:run

// lldb-command:expr self
// lldb-command:next
// lldb-command:expr self
// lldb-command:print $0 == $1
// lldb-check:true

struct Foo {
    a: usize,
}

impl Foo {
    #[inline(never)]
    fn get_a(&self) -> Option<usize> {
        Some(self.a)
    }

    #[inline(never)]
    fn var_return(&self) -> usize {
        let r = self.get_a().unwrap();
        r
    }

    #[inline(never)]
    fn var_return_opt_unwrap(&self) -> Option<usize> {
        let r = self.get_a().unwrap();
        Some(r)
    }

    #[inline(never)]
    fn var_return_opt_match(&self) -> Option<usize> {
        let r = match self.get_a() {
            None => return None,
            Some(a) => a,
        };
        Some(r)
    }

    #[inline(never)]
    fn var_return_opt_try(&self) -> Option<usize> {
        let r = self.get_a()?;
        Some(r)
    }
}

fn main() {
    let f1 = Foo{ a: 1 };
    let f2 = Foo{ a: 1 };
    f1.var_return();
    f1.var_return_opt_unwrap();
    f1.var_return_opt_match();
    f2.var_return_opt_try();
}
