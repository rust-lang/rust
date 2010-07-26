fn main() {
  let char yen = '¥';         // 0xa5
  let char c_cedilla = 'ç';   // 0xe7
  let char thorn = 'þ';       // 0xfe
  let char y_diaeresis = 'ÿ'; // 0xff
  let char pi = 'Π';          // 0x3a0

  check ((yen as int) == 0xa5);
  check ((c_cedilla as int) == 0xe7);
  check ((thorn as int) == 0xfe);
  check ((y_diaeresis as int) == 0xff);
  check ((pi as int) == 0x3a0);

  check ((pi as int) == ('\u03a0' as int));
  check (('\x0a' as int) == ('\n' as int));

  let str bhutan = "འབྲུག་ཡུལ།";
  let str japan = "日本";
  let str uzbekistan = "Ўзбекистон";
  let str austria = "Österreich";

  let str bhutan_e =
    "\u0f60\u0f56\u0fb2\u0f74\u0f42\u0f0b\u0f61\u0f74\u0f63\u0f0d";
  let str japan_e = "\u65e5\u672c";
  let str uzbekistan_e =
    "\u040e\u0437\u0431\u0435\u043a\u0438\u0441\u0442\u043e\u043d";
  let str austria_e = "\u00d6sterreich";

  let char oo = 'Ö';
  check ((oo as int) == 0xd6);

  fn check_str_eq(str a, str b) {
    let int i = 0;
    for (u8 ab in a) {
      log i;
      log ab;
      let u8 bb = b.(i);
      log bb;
      check(ab == bb);
      i += 1;
    }
  }

  check_str_eq(bhutan, bhutan_e);
  check_str_eq(japan, japan_e);
  check_str_eq(uzbekistan, uzbekistan_e);
  check_str_eq(austria, austria_e);
}
