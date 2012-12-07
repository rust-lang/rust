extern mod std;

fn main() {
    let x = std::map::HashMap();
    x.insert((@"abc", 0), 0);
}
