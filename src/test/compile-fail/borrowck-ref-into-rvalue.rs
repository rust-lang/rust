fn main() {
    let msg;
    match some(~"Hello") { //~ ERROR illegal borrow
        some(ref m) => {
            msg = m;
        },  
        none => { fail }
    }   
    io::println(*msg);
}

