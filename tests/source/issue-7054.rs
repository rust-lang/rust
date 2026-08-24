// rustfmt-edition: 2024

// `for await` is spelled with two tokens, so the source may separate them with
// arbitrary whitespace or comments. rustfmt must not search for the rendered
// keyword `for await` as a single literal string.

#![feature(async_iterator, async_for_loop)]

async fn for_await_canonical(iter: Iter) {
    for await i in iter {}
}

async fn for_await_extra_spaces(iter: Iter) {
    for   await i in iter {}
}

async fn for_await_newline(iter: Iter) {
    for
        await i in iter {}
}

async fn for_await_comment_between_keywords(iter: Iter) {
    for /* between for and await */ await i in iter {}
}

async fn for_await_comment_after_keyword(iter: Iter) {
    for await /* between await and pat */ i in iter {}
}

async fn for_await_comment_both_gaps(iter: Iter) {
    for /* first gap */ await /* second gap */ i in iter {}
}

async fn for_await_labeled(iter: Iter) {
    'outer: for   await i in iter {}
}

async fn for_await_body(iter: Iter) {
    for await i in iter {
        do_something(i);
    }
}

// Single-token keywords must be unaffected by the change.

fn plain_for(iter: Iter) {
    for   i in iter {}
    for /* comment */ i in iter {}
    'outer: for i in iter {}
}

fn plain_while(cond: bool) {
    while   cond {}
    while /* comment */ cond {}
    'outer: while cond {}
}

fn plain_while_let(opt: Opt) {
    while   let Some(x) = opt {}
    while /* comment */ let Some(x) = opt {}
}

fn plain_loop() {
    loop {}
    'outer: loop {}
}

fn plain_if(cond: bool) {
    if   cond {}
    if /* comment */ cond {}
    if   let Some(x) = opt {}
}
