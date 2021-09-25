// only-aarch64

#![feature(asm)]

macro_rules! m {
    ($in:ident $out:ident $lateout:ident $inout:ident $inlateout:ident $const:ident $sym:ident
     $pure:ident $nomem:ident $readonly:ident $preserves_flags:ident
     $noreturn:ident $nostack:ident $options:ident) => {
        unsafe {
            asm!("", $in(x) x, $out(x) x, $lateout(x) x, $inout(x) x, $inlateout(x) x,
            //~^ ERROR asm outputs are not allowed with the `noreturn` option
            const x, sym x,
            $options($pure, $nomem, $readonly, $preserves_flags, $noreturn, $nostack));
            //~^ ERROR the `nomem` and `readonly` options are mutually exclusive
            //~| ERROR the `pure` and `noreturn` options are mutually exclusive
        }
    };
}

fn main() {
    m!(in out lateout inout inlateout const sym
       pure nomem readonly preserves_flags
       noreturn nostack options);
}
