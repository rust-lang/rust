module.exports = {
    env: {
        es6: true,
        node: true,
    },
    extends: ["prettier"],
    parser: "@typescript-eslint/parser",
    parserOptions: {
        project: "tsconfig.eslint.json",
        tsconfigRootDir: __dirname,
        sourceType: "module",
    },
    plugins: ["@typescript-eslint"],
    rules: {
        camelcase: ["error"],
        eqeqeq: ["error", "always", { null: "ignore" }],
        curly: ["error", "multi-line"],
        "no-console": ["error", { allow: ["warn", "error"] }],
        "prefer-const": "error",
        "@typescript-eslint/member-delimiter-style": [
            "error",
            {
                multiline: {
                    delimiter: "semi",
                    requireLast: true,
                },
                singleline: {
                    delimiter: "semi",
                    requireLast: false,
                },
            },
        ],
        "@typescript-eslint/semi": ["error", "always"],
        "@typescript-eslint/no-unnecessary-type-assertion": "error",
        "@typescript-eslint/no-floating-promises": "error",
    },
};
