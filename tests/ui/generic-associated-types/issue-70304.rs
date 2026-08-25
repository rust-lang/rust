trait Document {
    type Cursor<'a>: DocCursor<'a>;
    //~^ ERROR: missing required bound on `Cursor`

    fn cursor(&self) -> Self::Cursor<'_>;
}

struct DocumentImpl {}

impl Document for DocumentImpl {
    type Cursor<'a> = DocCursorImpl<'a>;

    fn cursor(&self) -> Self::Cursor<'_> {
        DocCursorImpl { document: &self }
    }
}

trait DocCursor<'a> {}

struct DocCursorImpl<'a> {
    document: &'a DocumentImpl,
}

impl<'a> DocCursor<'a> for DocCursorImpl<'a> {}

struct Lexer<'d, Cursor>
where
    Cursor: DocCursor<'d>,
{
    cursor: Cursor,
    _phantom: std::marker::PhantomData<&'d ()>,
}

impl<'d, Cursor> Lexer<'d, Cursor>
where
    Cursor: DocCursor<'d>,
{
    pub fn from<Doc>(document: &'d Doc) -> Lexer<'d, Cursor>
    where
        Doc: Document<Cursor<'d> = Cursor>,
    {
        Lexer { cursor: document.cursor(), _phantom: std::marker::PhantomData }
    }
}

fn create_doc() -> impl Document<Cursor<'_> = DocCursorImpl<'_>> {
    //~^ ERROR `'_` cannot be used here [E0637]
    //~| ERROR: missing lifetime specifier
    DocumentImpl {}
}

pub fn main() {
    let doc = create_doc();
    let lexer: Lexer<'_, DocCursorImpl<'_>> = Lexer::from(&doc);
}
