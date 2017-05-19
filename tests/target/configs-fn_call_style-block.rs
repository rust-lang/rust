// rustfmt-fn_call_style: Block
// Function call style

fn main() {
    lorem(
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
    );
    // #1501
    let hyper = Arc::new(
        Client::with_connector(HttpsConnector::new(TlsClient::new())),
    );
}

// #1521
impl Foo {
    fn map_pixel_to_coords(&self, point: &Vector2i, view: &View) -> Vector2f {
        unsafe {
            Vector2f::from_raw(
                ffi::sfRenderTexture_mapPixelToCoords(self.render_texture, point.raw(), view.raw()),
            )
        }
    }
}

fn issue1420() {
    given(
        r#"
        # Getting started
        ...
    "#
    )
        .running(waltz)
}

// #1563
fn query(conn: &Connection) -> Result<()> {
    conn.query_row(
        r#"
            SELECT title, date
            FROM posts,
            WHERE DATE(date) = $1
        "#,
        &[],
        |row| {
            Post {
                title: row.get(0),
                date: row.get(1),
            }
        },
    )?;

    Ok(())
}
