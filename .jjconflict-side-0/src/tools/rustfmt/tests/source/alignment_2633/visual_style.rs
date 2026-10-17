// rustfmt-struct_field_align_threshold: 50
// rustfmt-indent_style: Visual

fn func() {
    Ok(ServerInformation { name:         unwrap_message_string(items.get(0)),
           vendor: unwrap_message_string(items.get(1)),
           version: unwrap_message_string(items.get(2)),
           spec_version: unwrap_message_string(items.get(3)), });
}
