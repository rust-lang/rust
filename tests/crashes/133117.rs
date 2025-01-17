//@ known-bug: #133117

fn main() {
    match () {
        (!|!) if true => {}
        (!|!) if true => {}
    }
}
