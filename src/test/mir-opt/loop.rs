// EMIT_MIR loop.f.SimplifyCfg-final.diff

fn f(){
    loop{}
}

fn main() {
    f();
}
