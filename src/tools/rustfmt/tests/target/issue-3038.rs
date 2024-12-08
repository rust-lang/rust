impl HTMLTableElement {
    fn func() {
        if number_of_row_elements == 0 {
            if let Some(last_tbody) = node
                .rev_children()
                .filter_map(DomRoot::downcast::<Element>)
                .find(|n| {
                    n.is::<HTMLTableSectionElement>() && n.local_name() == &local_name!("tbody")
                })
            {
                last_tbody
                    .upcast::<Node>()
                    .AppendChild(new_row.upcast::<Node>())
                    .expect("InsertRow failed to append first row.");
            }
        }

        if number_of_row_elements == 0 {
            if let Some(last_tbody) = node.find(|n| {
                n.is::<HTMLTableSectionElement>() && n.local_name() == &local_name!("tbody")
            }) {
                last_tbody
                    .upcast::<Node>()
                    .AppendChild(new_row.upcast::<Node>())
                    .expect("InsertRow failed to append first row.");
            }
        }
    }
}
