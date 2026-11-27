import { type Config } from "prettier";

const config: Config = {
    // use 4 because it's Rustfmt's default
    // https://rust-lang.github.io/rustfmt/?version=v1.4.38&search=#%5C34%20%5C%20%5C(default%5C)%5C%3A
    tabWidth: 4,
    // use 100 because it's Rustfmt's default
    // https://rust-lang.github.io/rustfmt/?version=v1.4.38&search=#max_width
    printWidth: 100,
};

export default config;
