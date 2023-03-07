use std::marker::PhantomData;

pub struct Directed;
pub struct Undirected;

pub struct Graph<N, E, Ty = Directed> {
    nodes: Vec<PhantomData<N>>,
    edges: Vec<PhantomData<E>>,
    ty: PhantomData<Ty>,
}


impl<N, E> Graph<N, E, Directed> {
    pub fn new() -> Self {
        Graph{nodes: Vec::new(), edges: Vec::new(), ty: PhantomData}
    }
}

impl<N, E> Graph<N, E, Undirected> {
    pub fn new_undirected() -> Self {
        Graph{nodes: Vec::new(), edges: Vec::new(), ty: PhantomData}
    }
}
