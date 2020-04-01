struct Pizza<T>(T);
struct Pineapple;

fn main() {
    let _: Pizza<Pineapple>; //~ERROR pineapple doesn't go on pizza
}
