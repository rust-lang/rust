// rustfmt-chains_overflow_last: false

fn main() {
    reader.lines()
        .map(|line| line.expect("Failed getting line"))
        .take_while(|line| line_regex.is_match(&line))
        .filter_map(|line| {
            regex.captures_iter(&line)
                .next()
                .map(|capture| {
                    (capture.at(1).expect("Couldn\'t unwrap capture").to_owned(),
                     capture.at(2).expect("Couldn\'t unwrap capture").to_owned())
                })
        })
        .collect();
}
