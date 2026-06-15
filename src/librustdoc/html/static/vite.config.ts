import type { UserConfig } from "vite-plus";

export default {
    fmt: {
        tabWidth: 4,
        printWidth: 100,
        arrowParens: "avoid",
        ignorePatterns: ["js/*.d.ts", "js/search.js", "js/src-script.js"],
    },
    lint: {
        plugins: [],
        categories: {
            correctness: "error",
        },
        env: {
            browser: true,
            es6: true,
        },
        rules: {
            "block-scoped-var": "error",
            eqeqeq: "error",
            "guard-for-in": "error",
            "no-alert": "error",
            "no-console": "error",
            "no-const-assign": "error",
            "no-debugger": "error",
            "no-div-regex": "error",
            "no-dupe-else-if": "error",
            "no-dupe-keys": "error",
            "no-duplicate-case": "error",
            "no-ex-assign": "error",
            "no-fallthrough": "error",
            "no-implicit-globals": "off",
            "no-implied-eval": "error",
            "no-import-assign": "error",
            "no-invalid-regexp": "error",
            "no-label-var": "error",
            "no-lonely-if": "error",
            "no-multi-assign": "error",
            "no-return-assign": "error",
            "no-script-url": "error",
            "no-sequences": "error",
            "no-self-compare": "error",
            "no-template-curly-in-string": "error",
            "no-unused-vars": [
                "error",
                {
                    argsIgnorePattern: "^_",
                    varsIgnorePattern: "^_",
                },
            ],
            "no-var": "error",
            "prefer-arrow-callback": "error",
            "prefer-const": "error",
        },
    },
} satisfies UserConfig;
