struct trie_node {
    mut content: ~[~str],
    mut children: ~[trie_node],
}

fn print_str_vector(vector: ~[~str]) {
    for vector.each() |string| {
        io::println(*string);
    }
}

fn main() {
    let node: trie_node = trie_node {
        content: ~[],
        children: ~[]
    };
    let v = ~[~"123", ~"abc"];
    node.content = ~[~"123", ~"abc"];
    print_str_vector(v);
    print_str_vector(copy node.content);

}
