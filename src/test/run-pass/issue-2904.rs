/// Map representation

use std;

enum square {
    bot,
    wall,
    rock,
    lambda,
    closed_lift,
    open_lift,
    earth,
    empty
}

impl square: to_str::ToStr {
    fn to_str() -> ~str {
        match self {
          bot => { ~"R" }
          wall => { ~"#" }
          rock => { ~"*" }
          lambda => { ~"\\" }
          closed_lift => { ~"L" }
          open_lift => { ~"O" }
          earth => { ~"." }
          empty => { ~" " } 
        }
    }
}

fn square_from_char(c: char) -> square {
    match c  {
      'R'  => { bot }
      '#'  => { wall }
      '*'  => { rock }
      '\\' => { lambda }
      'L'  => { closed_lift }
      'O'  => { open_lift }
      '.'  => { earth }
      ' '  => { empty }
      _ => {
        #error("invalid square: %?", c);
        fail
      }
    }
}

fn read_board_grid<rdr: owned io::Reader>(+in: rdr) -> ~[~[square]] {
    let in = in as io::Reader;
    let mut grid = ~[];
    for in.each_line |line| {
        let mut row = ~[];
        for line.each_char |c| {
            vec::push(row, square_from_char(c))
        }
        vec::push(grid, row)
    }
    let width = grid[0].len();
    for grid.each |row| { assert row.len() == width }
    grid
}

mod test {
    #[test]
    fn trivial_to_str() {
        assert lambda.to_str() == "\\"
    }

    #[test]
    fn read_simple_board() {
        let s = #include_str("./maps/contest1.map");
        io::with_str_reader(s, read_board_grid)
    }
}

fn main() {}
