// rustfmt-indent_style: Block
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
    let hyper = Arc::new(Client::with_connector(
        HttpsConnector::new(TlsClient::new()),
    ));

    // chain
    let x = yooooooooooooo
        .fooooooooooooooo
        .baaaaaaaaaaaaar(hello, world);

    // #1380
    {
        {
            let creds = self
                .client
                .client_credentials(&self.config.auth.oauth2.id, &self.config.auth.oauth2.secret)?;
        }
    }

    // nesting macro and function call
    try!(foo(
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ));
    try!(foo(try!(
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    )));
}

// #1521
impl Foo {
    fn map_pixel_to_coords(&self, point: &Vector2i, view: &View) -> Vector2f {
        unsafe {
            Vector2f::from_raw(ffi::sfRenderTexture_mapPixelToCoords(
                self.render_texture,
                point.raw(),
                view.raw(),
            ))
        }
    }
}

fn issue1420() {
    given(
        r#"
        # Getting started
        ...
    "#,
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
        |row| Post {
            title: row.get(0),
            date: row.get(1),
        },
    )?;

    Ok(())
}

// #1449
fn future_rayon_wait_1_thread() {
    // run with only 1 worker thread; this would deadlock if we couldn't make progress
    let mut result = None;
    ThreadPool::new(Configuration::new().num_threads(1))
        .unwrap()
        .install(|| {
            scope(|s| {
                use std::sync::mpsc::channel;
                let (tx, rx) = channel();
                let a = s.spawn_future(lazy(move || Ok::<usize, ()>(rx.recv().unwrap())));
                //                          ^^^^ FIXME: why is this needed?
                let b = s.spawn_future(a.map(|v| v + 1));
                let c = s.spawn_future(b.map(|v| v + 1));
                s.spawn(move |_| tx.send(20).unwrap());
                result = Some(c.rayon_wait().unwrap());
            });
        });
    assert_eq!(result, Some(22));
}

// #1494
impl Cursor {
    fn foo() {
        self.cur_type()
            .num_template_args()
            .or_else(|| {
                let n: c_int = unsafe { clang_Cursor_getNumTemplateArguments(self.x) };

                if n >= 0 {
                    Some(n as u32)
                } else {
                    debug_assert_eq!(n, -1);
                    None
                }
            })
            .or_else(|| {
                let canonical = self.canonical();
                if canonical != *self {
                    canonical.num_template_args()
                } else {
                    None
                }
            });
    }
}

fn issue1581() {
    bootstrap.checks.register("PERSISTED_LOCATIONS", move || {
        if locations2.0.inner_mut.lock().poisoned {
            Check::new(
                State::Error,
                "Persisted location storage is poisoned due to a write failure",
            )
        } else {
            Check::new(State::Healthy, "Persisted location storage is healthy")
        }
    });
}

fn issue1651() {
    {
        let type_list: Vec<_> =
            try_opt!(types.iter().map(|ty| ty.rewrite(context, shape)).collect());
    }
}
