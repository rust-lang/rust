trait Node {
    fn zomg();
}

trait Graph<N: Node> {
    fn nodes<'a, I: Iterator<Item=&'a N>>(&'a self) -> I
        where N: 'a;
}

impl<N: Node> Graph<N> for Vec<N> {
    fn nodes<'a, I: Iterator<Item=&'a N>>(&self) -> I
        where N: 'a
    {
        self.iter() //~ ERROR mismatched types
    }
}

struct Stuff;

impl Node for Stuff {
    fn zomg() {
        println!("zomg");
    }
}

fn iterate<N: Node, G: Graph<N>>(graph: &G) {
    for node in graph.iter() { //~ ERROR no method named `iter` found
        node.zomg(); //~ ERROR type annotations needed
    }
}

pub fn main() {
    let graph = Vec::new();

    graph.push(Stuff);

    iterate(graph); //~ ERROR mismatched types
}
