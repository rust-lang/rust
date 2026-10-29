pub struct Cache(
    RefCell<HashMap<
        TypeId,
        Box<@ Any>,
    >>
);

