fn main(/*
    ---
*/) {
    let x /* this is one line */ = 3;

    let x /*
           * this
           * is
           * multiple
           * lines
           */ = 3;

    let x = /*
           * this
           * is
           * multiple
           * lines
           * after
           * the
           * =
           */ 3;

    let x /*
           * this
           * is
           * multiple
           * lines
           * including
           * a

           * blank
           * line
           */ = 3;
}
