fn main() {
    // #1078
    let items = itemize_list(
        context.source_map,
        field_iter,
        "}",
        |item| match *item {
            StructLitField::Regular(ref field) => field.span.lo(),
            StructLitField::Base(ref expr) => {
                let last_field_hi = fields.last().map_or(span.lo(), |field| field.span.hi());
                let snippet = context.snippet(mk_sp(last_field_hi, expr.span.lo()));
                let pos = snippet.find_uncommented("..").unwrap();
                last_field_hi + BytePos(pos as u32)
            }
        },
        |item| match *item {
            StructLitField::Regular(ref field) => field.span.hi(),
            StructLitField::Base(ref expr) => expr.span.hi(),
        },
        |item| {
            match *item {
                StructLitField::Regular(ref field) => rewrite_field(
                    inner_context,
                    &field,
                    &Constraints::new(v_budget.checked_sub(1).unwrap_or(0), indent),
                ),
                StructLitField::Base(ref expr) => {
                    // 2 = ..
                    expr.rewrite(
                        inner_context,
                        &Constraints::new(try_opt!(v_budget.checked_sub(2)), indent + 2),
                    )
                    .map(|s| format!("..{}", s))
                }
            }
        },
        context.source_map.span_after(span, "{"),
        span.hi(),
    );

    // #1580
    self.0.pool.execute(move || {
        let _timer = segments.0.rotate_timer.time();
        if let Err(e) = segments.rotate_async(wal) {
            error!("error compacting segment storage WAL", unsafe { error: e.display() });
        }
    });

    // #1581
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
