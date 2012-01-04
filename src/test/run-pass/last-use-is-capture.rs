// Make sure #1399 stays fixed

fn main() {
    fn invoke(f: lambda()) { f(); }
    let k = ~22;
    let _u = {a: k};
    invoke {||log(error, k);}
}
