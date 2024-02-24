//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-pretty-loop.pp

pub fn foo(){
    loop{
        break;
    }
}
